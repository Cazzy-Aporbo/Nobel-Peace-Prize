import React from 'react'
import styled from 'styled-components'
import { theme } from '../theme'

const NavContainer = styled.nav`
  background: rgba(71, 105, 117, 0.3);
  padding: ${theme.spacing.lg};
  margin: ${theme.spacing.xl} auto;
  max-width: 1600px;
  border-radius: ${theme.borderRadius.md};
  border: 2px solid ${theme.colors.teal};
  box-shadow: ${theme.shadows.md};
`

const NavGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: ${theme.spacing.md};
  
  @media (max-width: ${theme.breakpoints.mobile}) {
    grid-template-columns: 1fr;
  }
`

const NavButton = styled.button`
  background: ${props => props.active 
    ? `linear-gradient(135deg, ${theme.colors.purple}, ${theme.colors.teal})`
    : 'rgba(87, 82, 196, 0.2)'
  };
  color: ${theme.colors.light};
  padding: ${theme.spacing.md};
  border-radius: ${theme.borderRadius.md};
  border-left: 3px solid ${props => props.active ? theme.colors.mint : theme.colors.purple};
  font-size: 1rem;
  font-weight: 600;
  text-align: left;
  transition: all ${theme.transitions.normal};
  
  &:hover {
    background: linear-gradient(135deg, ${theme.colors.purple}, ${theme.colors.teal});
    transform: translateX(5px);
    border-left-color: ${theme.colors.mint};
  }
`

const Navigation = ({ activeSection, setActiveSection }) => {
  const sections = [
    { id: 'overview', label: 'Executive Overview' },
    { id: 'temporal', label: 'Temporal Evolution' },
    { id: 'gender', label: 'Gender Disparity Analysis' },
    { id: 'geospatial', label: 'Geospatial Clustering' },
    { id: 'collaboration', label: 'Collaboration Networks' },
    { id: 'nlp', label: 'Natural Language Processing' },
    { id: 'ml', label: 'Machine Learning Models' },
    { id: 'statistics', label: 'Statistical Inference' }
  ]

  return (
    <NavContainer>
      <NavGrid>
        {sections.map(section => (
          <NavButton
            key={section.id}
            active={activeSection === section.id}
            onClick={() => setActiveSection(section.id)}
          >
            {section.label}
          </NavButton>
        ))}
      </NavGrid>
    </NavContainer>
  )
}

export default Navigation
